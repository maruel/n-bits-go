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

func mainImpl() error {
	hf_token := flag.String("hf-token", "", "HuggingFace token")
	hf_repo := flag.String("hf-repo", "", "HuggingFace repository, e.g. \"meta-llama/Llama-3.2-1B\"")
	flag.Parse()

	ctx := context.Background()
	hf, err := huggingface.New(*hf_token, "")
	if err != nil {
		return err
	}
	if *hf_repo != "" {
		p, err := hf.EnsureFile(ctx, huggingface.PackedFileRef("hf:"+*hf_repo+"/HEAD/model.safetensors"), 0o666)
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
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "n-bits: %s\n", err)
		os.Exit(1)
	}
}
