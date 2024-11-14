// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"testing"
)

func TestCmdAnalyze(t *testing.T) {
	// Load live a relatively small (151MiB) model.
	if err := cmdAnalyze(context.Background(), "", "openai", "whisper-tiny", "", ""); err != nil {
		t.Fatal(err)
	}
}
