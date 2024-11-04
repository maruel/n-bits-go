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
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/sillybot/huggingface"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
	"github.com/nlpodyssey/safetensors"
)

func effective(l []int) int {
	o := 0
	for _, v := range l {
		if v != 0 {
			o++
		}
	}
	return o
}

func calcBF16Histogram(t safetensors.TensorView) ([]int, []int, []int) {
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
	b0 := data[0]
	b1 := data[1]
	sign := (b1 & 0x80) >> 7
	exponent := ((b1 & 0x7F) << 1) | ((b0 & 0x80) >> 7)
	mantissa := b0 & 0x7F
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
		tensors := s.Tensors()
		maxNameLen := 0
		maxSizeLen := 0
		for _, tensor := range tensors {
			if l := len(tensor.Name); l > maxNameLen {
				maxNameLen = l
			}
			if l := len(strconv.Itoa(int(tensor.TensorView.DataLen() / 2))); l > maxSizeLen {
				maxSizeLen = l
			}
		}
		totalBytes := 0
		bytesWasted := 0
		for _, tensor := range tensors {
			t := tensor.TensorView
			name := tensor.Name
			if dt := t.DType(); dt != safetensors.BF16 {
				return fmt.Errorf("%s: can't handle dtype %s", name, dt)
			}
			numEl := int(t.DataLen() / 2)
			signs, exponents, mantissas := calcBF16Histogram(t)
			e_signs := effective(signs)
			e_exponents := effective(exponents)
			e_mantissas := effective(mantissas)
			b_signs := math.Log2(float64(e_signs))
			b_exponents := math.Log2(float64(e_exponents))
			b_mantissas := math.Log2(float64(e_mantissas))
			wasted := 1 - int(math.Ceil(b_signs)) + 8 - int(math.Ceil(b_exponents)) + 7 - int(math.Ceil(b_mantissas))
			fmt.Printf("%-*s (%*d): sign=%d/%d(%5.1f%%) %3.1fbits  exponent=%3d/%d(%5.1f%%) %3.1fbits  mantissa=%3d/%d(%5.1f%%) %3.1fbits  wasted=%dbits\n",
				maxNameLen, name, maxSizeLen, numEl,
				e_signs, 1<<1, float64(e_signs)/float64(1<<1)*100., b_signs,
				e_exponents, 1<<8, float64(e_exponents)/float64(1<<8)*100., b_exponents,
				e_mantissas, 1<<7, float64(e_mantissas)/float64(1<<7)*100., b_mantissas,
				wasted,
			)
			bytesWasted += wasted * numEl / 8
			totalBytes += numEl * 2
		}
		fmt.Printf("%d bytes (%.1f%%) wasted on %d bytes total\n", bytesWasted, 100.*float64(bytesWasted)/float64(totalBytes), totalBytes)
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
