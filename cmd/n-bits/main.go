// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/sillybot/huggingface"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
	"github.com/nlpodyssey/safetensors"
)

func calcHistogram(t safetensors.TensorView) ([]int, []int, []int) {
	data := t.Data()
	signs := [1 << 1]int{}
	exponents := [1 << 8]int{}
	mantissas := [1 << 7]int{}
	for i := range t.DataLen() / 2 {
		sign, exponent, mantissa := unpackBFloat16(data[2*i:])
		signs[sign]++
		exponents[exponent]++
		mantissas[mantissa]++
	}
	return signs[:], exponents[:], mantissas[:]
}

func unpackBFloat16(data []byte) (int, int, int) {
	// Probably a bit faster than proper unpacking via encoding/binary.
	b1 := data[1]
	b2 := data[0]
	sign := (b1 & 0x80) >> 7
	exponent := ((b1 & 0x7F) << 1) | ((b2 & 0x80) >> 7)
	mantissa := b2 & 0x7F
	return int(sign), int(exponent), int(mantissa)
}

func run(ctx context.Context, hfToken, hfRepo string) error {
	// TODO: The library kinda sucks for downloading and caching.
	cache := filepath.Join(".cache", hfRepo)
	if err := os.MkdirAll(cache, 0o777); err != nil {
		return err
	}
	hf, err := huggingface.New(hfToken, cache)
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
		//fmt.Printf("len = %d\n", s.Len())
		//fmt.Printf("names = %+v\n", s.Names())
		tensors := s.Tensors()
		for i := range 3 {
			t := tensors[i].TensorView
			name := tensors[i].Name
			if dt := t.DType(); dt != safetensors.BF16 {
				return fmt.Errorf("%s: can't handle dtype %s", name, dt)
			}
			fmt.Printf("Tensor %s\n", name)
			signs, exponents, mantissas := calcHistogram(t)
			fmt.Printf("- signs = %+v\n", signs)
			fmt.Printf("- exponents = %+v\n", exponents)
			fmt.Printf("- mantissas = %+v\n", mantissas)
		}
	}
	return nil
}

func mainImpl() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()
	programLevel := &slog.LevelVar{}
	programLevel.Set(slog.LevelError)
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      programLevel,
		TimeFormat: "15:04:05.000", // Like time.TimeOnly plus milliseconds.
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch t := a.Value.Any().(type) {
			case string:
				if t == "" {
					return slog.Attr{}
				}
			case bool:
				if !t {
					return slog.Attr{}
				}
			case uint64:
				if t == 0 {
					return slog.Attr{}
				}
			case int64:
				if t == 0 {
					return slog.Attr{}
				}
			case float64:
				if t == 0 {
					return slog.Attr{}
				}
			case time.Time:
				if t.IsZero() {
					return slog.Attr{}
				}
			case time.Duration:
				if t == 0 {
					return slog.Attr{}
				}
			}
			return a
		},
	}))
	slog.SetDefault(logger)
	go func() {
		<-ctx.Done()
		slog.Info("main", "message", "quitting")
	}()

	hfToken := flag.String("hf-token", "", "HuggingFace token")
	hfRepo := flag.String("hf-repo", "", "HuggingFace repository, e.g. \"meta-llama/Llama-3.2-1B\"")
	verbose := flag.Bool("v", false, "Enable verbose logging")
	flag.Parse()
	if len(flag.Args()) != 0 {
		return errors.New("unexpected argument")
	}
	if *verbose {
		programLevel.Set(slog.LevelDebug)
	}
	return run(ctx, *hfToken, *hfRepo)
}

func main() {
	if err := mainImpl(); err != nil {
		fmt.Fprintf(os.Stderr, "n-bits: %s\n", err)
		os.Exit(1)
	}
}
