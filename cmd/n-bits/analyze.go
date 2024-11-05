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
	"strconv"

	"github.com/maruel/huggingface"
	"github.com/maruel/n-bits-go/n_bits"
	"github.com/maruel/safetensors"
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

func analyze(ctx context.Context, hfToken, author, repo, out string) error {
	hf, err := huggingface.New(hfToken)
	if err != nil {
		return err
	}
	if repo != "" {
		ref := huggingface.ModelRef{Author: author, Repo: repo}
		files, err := hf.EnsureSnapshot(ctx, ref, "main", []string{"model*.safetensors"})
		if err != nil {
			return err
		}
		var totalBytes, bytesWasted int64
		all := n_bits.AnalyzedModel{}
		for _, f := range files {
			fmt.Printf("Processing %s:\n", filepath.Base(f))
			b, err := os.ReadFile(f)
			if err != nil {
				return err
			}
			s, err := safetensors.Deserialize(b)
			if err != nil {
				return err
			}
			tensors := s.Tensors()
			maxNameLen := 0
			maxSizeLen := 0
			analyzed := make([]n_bits.AnalyzedTensor, len(tensors))
			for i, tensor := range tensors {
				if l := len(tensor.Name); l > maxNameLen {
					maxNameLen = l
				}
				if l := len(strconv.Itoa(int(tensor.TensorView.DataLen() / 2))); l > maxSizeLen {
					maxSizeLen = l
				}
				if analyzed[i], err = n_bits.AnalyzeTensor(tensor.Name, tensor.TensorView); err != nil {
					return err
				}
			}
			for _, a := range analyzed {
				wasted := int64(a.Sign.BitsWasted() + a.Exponent.BitsWasted() + a.Mantissa.BitsWasted())
				fmt.Printf("%-*s: %*dw  avg=%4.1f [%6.1f, %6.1f]  sign=%1.0fbit  exponent=%3.1f/%dbits  mantissa=%3.1f/%dbits  wasted=%d/16bits %.1f%%  %8s\n",
					maxNameLen, a.Name, maxSizeLen, a.NumEl,
					a.Avg, a.Min, a.Max,
					a.Sign.BitsActuallyUsed(),
					a.Exponent.BitsActuallyUsed(), a.Exponent.Allocation,
					a.Mantissa.BitsActuallyUsed(), a.Mantissa.Allocation,
					wasted, 100.*float64(wasted)/16., humanBytes(wasted*a.NumEl/8),
				)
				bytesWasted += wasted * a.NumEl / 8
				totalBytes += a.NumEl * 2
			}
			all.Tensors = append(all.Tensors, analyzed...)
		}
		fmt.Printf("%d bytes (%.1f%%) wasted on %d bytes total\n", bytesWasted, 100.*float64(bytesWasted)/float64(totalBytes), totalBytes)
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
