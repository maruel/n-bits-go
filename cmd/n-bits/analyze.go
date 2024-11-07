// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"

	"github.com/maruel/huggingface"
	"github.com/maruel/n-bits-go/n_bits"
	"github.com/maruel/safetensors"
	"github.com/pbnjay/memory"
	"golang.org/x/sync/errgroup"
)

func humanBytes(i int64) string {
	switch {
	case i > 1024*1024*1024:
		return fmt.Sprintf("%.1fGiB", float64(i)/1024./1024./1024.)
	case i > 1024*1024:
		return fmt.Sprintf("%.1fMiB", float64(i)/1024./1024.)
	case i > 1024:
		return fmt.Sprintf("%.1fkiB", float64(i)/1024.)
	default:
		return fmt.Sprintf("%dB", i)
	}
}

func processSafetensorsFile(ctx context.Context, name string, cpuLimit chan struct{}) ([]n_bits.AnalyzedTensor, error) {
	// TODO: Memory map instead of reading? Need to perf test.
	/*
		f, err := os.OpenFile(name, os.O_RDWR, 0o600)
		defer f.Close()
		// github.com/edsrzf/mmap-go
		mmap, err := mmap.Map(f, mmap.RDWR, 0)
		defer mmap.Unmap()
	*/
	b, err := os.ReadFile(name)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	s, err := safetensors.Deserialize(b)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	tensors := s.NamedTensors()
	analyzed := make([]n_bits.AnalyzedTensor, len(tensors))
	// Analyze tensors concurrently.
	eg := errgroup.Group{}
	for i, tensor := range tensors {
		eg.Go(func() error {
			cpuLimit <- struct{}{}
			defer func() {
				<-cpuLimit
			}()
			if err = ctx.Err(); err != nil {
				return err
			}
			var err2 error
			analyzed[i], err2 = n_bits.AnalyzeTensor(tensor.Name, tensor.TensorView)
			return err2
		})
	}
	err = eg.Wait()
	return analyzed, err
}

func calcNameLen(tensors []n_bits.AnalyzedTensor) (int, int) {
	maxNameLen := 0
	maxSizeLen := 0
	for _, tensor := range tensors {
		if l := len(tensor.Name); l > maxNameLen {
			maxNameLen = l
		}
		if l := len(strconv.FormatInt(tensor.NumEl, 10)); l > maxSizeLen {
			maxSizeLen = l
		}
	}
	return maxNameLen, maxSizeLen
}

func analyze(ctx context.Context, hfToken, author, repo, out string) error {
	hf, err := huggingface.New(hfToken)
	if err != nil {
		return err
	}
	if repo != "" {
		ref := huggingface.ModelRef{Author: author, Repo: repo}
		// FLUX.1-dev uses "flux1-dev.safetensors".
		files, err := hf.EnsureSnapshot(ctx, ref, "main", []string{"*.safetensors"})
		if err != nil {
			return err
		}

		mu := sync.Mutex{}
		all := n_bits.AnalyzedModel{}

		// Concurrency limit.
		cpuLimit := make(chan struct{}, runtime.NumCPU())
		// This is limited by the amount of RAM.
		// Assume roughly 4GiB per safetensors, round down, then minus one. In
		// practice safetensors tend to be about 4.5GiB.
		// TODO: limit by actual safetensors size. This is very approximative and
		// will lead to crashes.
		p := memory.TotalMemory()/1024/1024/1024/8 - 1
		if p <= 0 {
			p = 1
		}
		loadPipe := make(chan string, p)
		go func() {
			// TODO: Handle cancelation.
			for _, f := range files {
				loadPipe <- f
			}
			close(loadPipe)
		}()

		eg, ctx := errgroup.WithContext(ctx)
		for range p {
			eg.Go(func() error {
				// TODO: Use a pipeline so they are processed in order.
				for f := range loadPipe {
					if err2 := ctx.Err(); err2 != nil {
						return err2
					}
					// TODO: This prints stuff out of order.
					fmt.Printf("Processing %s:\n", filepath.Base(f))
					// TODO: os.Stat() the file and "consume" this amount of ram from the throttler.
					analyzed, err2 := processSafetensorsFile(ctx, f, cpuLimit)
					if err2 != nil {
						return err2
					}
					if err2 := ctx.Err(); err2 != nil {
						return err2
					}
					maxNameLen, maxSizeLen := calcNameLen(analyzed)
					for _, a := range analyzed {
						bits := 8 * a.DType.Size()
						ratio := 100. / float64(bits)
						wasted := int64(a.Sign.BitsWasted() + a.Exponent.BitsWasted() + a.Mantissa.BitsWasted())
						fmt.Printf("%-*s: %*dw  avg=%4.1f [%6.1f, %6.1f]  sign=%1.0fbit  exponent=%3.1f/%dbits  mantissa=%3.1f/%dbits  wasted=%d/%dbits %.1f%%  %8s\n",
							maxNameLen, a.Name, maxSizeLen, a.NumEl,
							a.Avg, a.Min, a.Max,
							a.Sign.BitsActuallyUsed(),
							a.Exponent.BitsActuallyUsed(), a.Exponent.Allocation,
							a.Mantissa.BitsActuallyUsed(), a.Mantissa.Allocation,
							wasted, bits, ratio*float64(wasted), humanBytes(wasted*a.NumEl/8),
						)
					}
					mu.Lock()
					all.Tensors = append(all.Tensors, analyzed...)
					mu.Unlock()
				}
				return nil
			})
		}
		if err = eg.Wait(); err != nil {
			return err
		}
		var bytesWasted, totalBytes, totalWeights int64
		for _, a := range all.Tensors {
			bytesWasted += a.NumEl * int64(a.Sign.BitsWasted()+a.Exponent.BitsWasted()+a.Mantissa.BitsWasted()) / 8
			totalBytes += a.Len()
			totalWeights += a.NumEl
		}
		fmt.Printf("%s (%.1f%%) wasted on %s total storing %d weights\n", humanBytes(bytesWasted), 100.*float64(bytesWasted)/float64(totalBytes), humanBytes(totalBytes), totalWeights)
		if out != "" {
			data, err := json.Marshal(all)
			if err != nil {
				return err
			}
			if err := os.WriteFile(out, data, 0o666); err != nil {
				return err
			}
		}
	}
	return nil
}
