// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"testing"
)

func TestRun(t *testing.T) {
	if err := run(context.Background(), "", "Qwen/Qwen2.5-0.5B"); err != nil {
		t.Fatal(err)
	}
}
