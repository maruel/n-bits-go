// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
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

func processSafetensorsFile(ctx context.Context, name string, reTensors *regexp.Regexp, cpuLimit chan struct{}) ([]n_bits.AnalyzedTensor, error) {
	s := safetensors.Mapped{}
	if err := s.Open(name); err != nil {
		return nil, err
	}
	defer s.Close()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	toAnalyze := make([]int, 0, len(s.Tensors))
	for i, tensor := range s.Tensors {
		if reTensors.MatchString(tensor.Name) {
			toAnalyze = append(toAnalyze, i)
		}
	}
	slog.Info("analyze", "file", filepath.Base(name), "num_tensors", len(s.Tensors), "to_analyze", len(toAnalyze))
	analyzed := make([]n_bits.AnalyzedTensor, len(toAnalyze))
	// Analyze tensors concurrently.
	eg := errgroup.Group{}
	for j, i := range toAnalyze {
		eg.Go(func() error {
			cpuLimit <- struct{}{}
			defer func() {
				<-cpuLimit
			}()
			if err2 := ctx.Err(); err2 != nil {
				return err2
			}
			var err2 error
			n := s.Tensors[i].Name
			slog.Info("analyze", "file", filepath.Base(name), "name", n, "dtype", s.Tensors[i].DType)
			analyzed[j], err2 = n_bits.AnalyzeTensor(n, s.Tensors[i])
			return err2
		})
	}
	err := eg.Wait()
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

func cmdAnalyze(ctx context.Context, hfToken, author, repo, fileglob string, reTensors *regexp.Regexp, out string) error {
	hf, err := huggingface.New(hfToken)
	if err != nil {
		return err
	}
	if repo != "" {
		if fileglob == "" {
			fileglob = "*.safetensors"
		}
		ref := huggingface.ModelRef{Author: author, Repo: repo}
		files, err := hf.EnsureSnapshot(ctx, ref, "main", []string{fileglob})
		if err != nil {
			return err
		}

		mu := sync.Mutex{}
		all := n_bits.AnalyzedModel{}

		// Concurrency limit.
		cpus := runtime.NumCPU()
		if cpus < 2 {
			cpus = 2
		} else if cpus > 1024 {
			// Limit for now.
			cpus = 1024
		}
		cpuLimit := make(chan struct{}, cpus)
		// This is limited by the amount of RAM.
		// Assume roughly 4GiB per safetensors, round down, then minus one. In
		// practice safetensors tend to be about 4.5GiB but there are exceptions.
		// TODO: limit by actual safetensors size. This is very approximative and
		// will lead to crashes.
		p := memory.TotalMemory()/1024/1024/1024/5 - 1
		if p < 1 {
			p = 1
		} else if p > 16 {
			// limit for now.
			p = 16
		}
		loadPipe := make(chan string, p)
		go func() {
			// TODO: Handle cancelation.
			for _, f := range files {
				loadPipe <- f
			}
			close(loadPipe)
		}()

		eg, ctx2 := errgroup.WithContext(ctx)
		for range p {
			eg.Go(func() error {
				// TODO: Use a pipeline so they are processed in order.
				for f := range loadPipe {
					if err2 := ctx2.Err(); err2 != nil {
						return err2
					}
					// TODO: This prints stuff out of order.
					fmt.Printf("Processing %s:\n", filepath.Base(f))
					// TODO: os.Stat() the file and "consume" this amount of ram from the throttler.
					analyzed, err2 := processSafetensorsFile(ctx2, f, reTensors, cpuLimit)
					if err2 != nil {
						return err2
					}
					if err2 := ctx2.Err(); err2 != nil {
						return err2
					}
					maxNameLen, maxSizeLen := calcNameLen(analyzed)
					for _, a := range analyzed {
						bits := 8 * a.DType.WordSize()
						ratio := 100. / float64(bits)
						wasted := int64(a.Sign.BitsWasted() + a.Exponent.BitsWasted() + a.Mantissa.BitsWasted())
						if a.Exponent.GetAllocation() != 0 {
							// Integers.
							fmt.Printf("%-*s: %*dw  avg=%4.1f [%6.1f, %6.1f]  sign=%1.0fbit  exponent=%3.1f/%dbits  mantissa=%4.1f/%dbits  wasted=%2d/%dbits %4.1f%%  %8s\n",
								maxNameLen, a.Name, maxSizeLen, a.NumEl,
								a.Avg, a.Min, a.Max,
								a.Sign.BitsActuallyUsed(),
								a.Exponent.BitsActuallyUsed(), a.Exponent.GetAllocation(),
								a.Mantissa.BitsActuallyUsed(), a.Mantissa.GetAllocation(),
								wasted, bits, ratio*float64(wasted), humanBytes(wasted*a.NumEl/8),
							)
						} else {
							fmt.Printf("%-*s: %*dw  avg=%11.0f [%11.0f, %10.0f]  sign=%1.0fbit  mantissa=%4.1f/%dbits  wasted=%2d/%dbits %4.1f%%  %8s\n",
								maxNameLen, a.Name, maxSizeLen, a.NumEl,
								a.Avg, a.Min, a.Max,
								a.Sign.BitsActuallyUsed(),
								a.Mantissa.BitsActuallyUsed(), a.Mantissa.GetAllocation(),
								wasted, bits, ratio*float64(wasted), humanBytes(wasted*a.NumEl/8),
							)
						}
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
