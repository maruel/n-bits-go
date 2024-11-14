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
	"strings"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

type hfTokenArg string

func (h *hfTokenArg) Set(s string) error {
	if !strings.HasPrefix(s, "hf_") {
		return errors.New("token is invalid")
	}
	*h = hfTokenArg(s)
	return nil
}

func (h *hfTokenArg) String() string {
	return string(*h)
}

type hfRepoArg string

func (h *hfRepoArg) Set(s string) error {
	parts := strings.Split(s, "/")
	if len(parts) != 2 || len(parts[0]) == 0 || len(parts[1]) == 0 {
		return errors.New("repo is invalid")
	}
	*h = hfRepoArg(s)
	return nil
}

func (h *hfRepoArg) String() string {
	return string(*h)
}

func (h *hfRepoArg) Org() string {
	if len(*h) == 0 {
		return ""
	}
	s := string(*h)
	i := strings.IndexByte(s, '/')
	return s[:i]
}

func (h *hfRepoArg) Repo() string {
	if len(*h) == 0 {
		return ""
	}
	s := string(*h)
	i := strings.IndexByte(s, '/')
	return s[i+1:]
}

func mainImpl(args []string) error {
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
	defer func() {
		if r := recover(); r != nil {
			slog.Error("main", "panic", r)
			panic(r)
		}
	}()

	fs := flag.NewFlagSet("n-bits", flag.ContinueOnError)
	verbose := fs.Bool("v", false, "Enable verbose logging")
	if len(args) == 0 {
		fs.Usage()
		return context.Canceled
	}
	switch args[0] {
	case "analyze":
		var hfToken hfTokenArg
		var hfRepo hfRepoArg
		fs.Var(&hfToken, "hf-token", "HuggingFace token")
		fs.Var(&hfRepo, "hf-repo", "HuggingFace repository, e.g. \"meta-llama/Llama-3.2-1B\"")
		hfGlob := fs.String("hf-glob", "", "Glob to use when loading files (default:*.safetensors)")
		out := fs.String("json", "", "Save stats as a JSON file")
		if fs.Parse(args[1:]) != nil {
			return context.Canceled
		}
		if len(fs.Args()) != 0 {
			return errors.New("unexpected argument")
		}
		if *verbose {
			programLevel.Set(slog.LevelDebug)
		}
		if hfRepo == "" {
			return errors.New("-hf-repo is required")
		}
		return cmdAnalyze(ctx, hfToken.String(), hfRepo.Org(), hfRepo.Repo(), *hfGlob, *out)
	case "metadata":
		var hfToken hfTokenArg
		var hfRepo hfRepoArg
		fs.Var(&hfToken, "hf-token", "HuggingFace token")
		fs.Var(&hfRepo, "hf-repo", "HuggingFace repository, e.g. \"meta-llama/Llama-3.2-1B\"")
		hfGlob := fs.String("hf-glob", "", "Glob to use when loading files (default:*.safetensors)")
		name := fs.String("name", "", "Single file to process")
		if fs.Parse(args[1:]) != nil {
			return context.Canceled
		}
		if len(fs.Args()) != 0 {
			return errors.New("unexpected argument")
		}
		if *verbose {
			programLevel.Set(slog.LevelDebug)
		}
		if *name == "" {
			if hfRepo == "" {
				return errors.New("-hf-repo is required")
			}
		} else {
			if hfToken != "" {
				return errors.New("can't use both -name and -hf-token")
			}
			if hfRepo != "" {
				return errors.New("can't use both -name and -hf-repo")
			}
			if *hfGlob != "" {
				return errors.New("can't use both -name and -hf-glob")
			}
		}
		return cmdMetadata(ctx, *name, hfToken.String(), hfRepo.Org(), hfRepo.Repo(), *hfGlob)
	default:
		fs.Usage()
		return context.Canceled
	}
}

func main() {
	if err := mainImpl(os.Args[1:]); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "n-bits: %s\n", err)
		}
		os.Exit(1)
	}
}
