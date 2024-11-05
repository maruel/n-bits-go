// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"text/template"

	"github.com/maruel/n-bits-go/floatx"
)

const srcTmpl = `// Code generated "go run gen.go" DO NOT EDIT.

package floatx_test

import "math"

// See floatx_test.go
var {{.Name}} = []testData16{
{{range .Data}}{ {{.V}}, {{.F}}, {{.Sign}}, {{.Exponent}}, {{.Mantissa}}, },
{{end}} }
`

type testData16 struct {
	V        string
	F        string
	Sign     uint8
	Exponent uint8
	Mantissa uint16
}

func genBF16() []testData16 {
	const (
		signOffset     = 15
		exponentOffset = 7
		exponentMask   = (1 << (signOffset - exponentOffset)) - 1
		mantissaMask   = (1 << exponentOffset) - 1
	)
	var out [1 << 16]testData16
	for i := range out {
		x := testData16{
			V:        fmt.Sprintf("0x%04x", i),
			Sign:     uint8(i >> signOffset),
			Exponent: uint8((i >> exponentOffset) & exponentMask),
			Mantissa: uint16(i & mantissaMask),
		}
		f := floatx.BF16(i).Float32()
		// There's a problem here with denormalized values
		if sign := int(x.Sign) * -1; math.IsInf(float64(f), sign) {
			x.F = fmt.Sprintf("float32(math.Inf(%d))", sign)
		} else if math.IsNaN(float64(f)) {
			x.F = "float32(math.NaN())"
		} else {
			x.F = fmt.Sprintf("%g", f)
		}
		out[i] = x
	}
	return out[:]
}

func genF16() []testData16 {
	const (
		signOffset     = 15
		exponentOffset = 10
		exponentMask   = (1 << (signOffset - exponentOffset)) - 1
		mantissaMask   = (1 << exponentOffset) - 1
	)
	var out [1 << 16]testData16
	for i := range out {
		x := testData16{
			V:        fmt.Sprintf("0x%04x", i),
			Sign:     uint8(i >> signOffset),
			Exponent: uint8((i >> exponentOffset) & exponentMask),
			Mantissa: uint16(i & mantissaMask),
		}
		f := floatx.F16(i).Float32()
		if sign := int(x.Sign) * -1; math.IsInf(float64(f), sign) {
			x.F = fmt.Sprintf("float32(math.Inf(%d))", sign)
		} else if math.IsNaN(float64(f)) {
			x.F = "float32(math.NaN())"
		} else {
			x.F = fmt.Sprintf("%g", f)
		}
		out[i] = x
	}
	return out[:]
}

func generate(name, filename string, td []testData16) {
	data := map[string]any{
		"Name": name,
		"Data": td,
	}
	f, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	t := template.Must(template.New("").Parse(srcTmpl))
	if err := t.Execute(f, data); err != nil {
		panic(err)
	}
	if err := exec.Command("gofmt", "-w", "-s", filename).Run(); err != nil {
		panic(fmt.Errorf("failed to run gofmt: %w", err))
	}
}

func main() {
	generate("bf16TestData", "bf16_data_test.go", genBF16())
	generate("f16TestData", "f16_data_test.go", genF16())
}