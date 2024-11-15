// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/maruel/huggingface"
	"github.com/maruel/safetensors"
)

func loadMetadata(name string) (*safetensors.Mapped, error) {
	s := &safetensors.Mapped{}
	if err := s.Open(name); err != nil {
		return nil, err
	}
	return s, nil
}

func cmdMetadata(ctx context.Context, name, hfToken, author, repo, fileglob string) error {
	hf, err := huggingface.New(hfToken)
	if err != nil {
		return err
	}
	var files []string
	if name != "" {
		files = []string{name}
	} else {
		if fileglob == "" {
			fileglob = "*.safetensors"
		}
		ref := huggingface.ModelRef{Author: author, Repo: repo}
		var err error
		files, err = hf.EnsureSnapshot(ctx, ref, "main", []string{fileglob})
		if err != nil {
			return err
		}
	}
	for _, f := range files {
		s, err := loadMetadata(f)
		if err != nil {
			return err
		}
		fmt.Printf("%s:\n", filepath.Base(f))
		types := map[safetensors.DType]int{}
		for _, t := range s.Tensors {
			types[t.DType]++
		}
		for dtype, count := range types {
			fmt.Printf("  %d tensors of type %s\n", count, dtype)
		}
		for k, v := range s.Metadata {
			fmt.Printf("- %s: %s\n", k, v)
		}
		s.Close()
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}
