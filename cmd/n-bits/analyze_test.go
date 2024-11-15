// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"regexp"
	"testing"
)

func TestCmdAnalyze(t *testing.T) {
	// Load live a relatively small (151MiB) model.
	reTensors, err := regexp.Compile(".*")
	if err != nil {
		t.Fatal(err)
	}
	if err := cmdAnalyze(context.Background(), "", "openai", "whisper-tiny", "", reTensors, ""); err != nil {
		t.Fatal(err)
	}
}
