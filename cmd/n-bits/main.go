// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/maruel/sillybot/huggingface"
	"github.com/nlpodyssey/safetensors"
)

func calcHistogram(t string) {
}

func run(ctx context.Context, hfToken, hfRepo string) error {
	hf, err := huggingface.New(hfToken, "")
	if err != nil {
		return err
	}
	if hfRepo != "" {
		p, err := hf.EnsureFile(ctx, huggingface.PackedFileRef("hf:"+hfRepo+"/HEAD/model.safetensors"), 0o666)
		if err != nil {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		s, err := safetensors.Deserialize(b)
		if err != nil {
			return err
		}
		fmt.Printf("len = %d\n", s.Len())
		fmt.Printf("names = %+v\n", s.Names())
		tensors := s.Tensors()
		for i := range 3 {
			t := tensors[i].TensorView
			if t.DType() != safetensors.BF16 {
				return fmt.Errorf("can't handle dtype %s", t.DType())
			}
			data := t.Data()
			signs := [1 << 1]int{}
			exponents := [1 << 8]int{}
			mantissas := [1 << 7]int{}
			for i := range t.DataLen() / 2 {
				// Probably a bit faster than proper unpacking via encoding/binary.
				b1 := data[2*i]
				b2 := data[2*i+1]
				sign := (b1 & 0x80) >> 7
				exponent := ((b1 & 0x7F) << 1) | ((b2 & 0x80) >> 7)
				mantissa := b2 & 0x7F
				signs[sign]++
				exponents[exponent]++
				mantissas[mantissa]++
			}
			fmt.Printf("Tensor %s\n", tensors[i].Name)
			fmt.Printf("- signs = %+v\n", signs)
			fmt.Printf("- exponents = %+v\n", exponents)
			fmt.Printf("- mantissas = %+v\n", mantissas)
		}
	}
	return nil
}

func mainImpl() error {
	hfToken := flag.String("hf-token", "", "HuggingFace token")
	hfRepo := flag.String("hf-repo", "", "HuggingFace repository, e.g. \"meta-llama/Llama-3.2-1B\"")
	flag.Parse()

	ctx := context.Background()
	return run(ctx, *hfToken, *hfRepo)
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "n-bits: %s\n", err)
		os.Exit(1)
	}
}
