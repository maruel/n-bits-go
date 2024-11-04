module github.com/maruel/n-bits-go

go 1.23.1

require (
	github.com/lmittmann/tint v1.0.5
	github.com/maruel/sillybot v0.0.0-20240930182418-fa372ed0296c
	github.com/mattn/go-colorable v0.1.13
	github.com/mattn/go-isatty v0.0.20
	github.com/nlpodyssey/safetensors v0.0.0-20230602165149-2a74a2d18984
)

require (
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/schollz/progressbar/v3 v3.17.0 // indirect
	golang.org/x/sys v0.26.0 // indirect
	golang.org/x/term v0.25.0 // indirect
)

// Necessary for determinism.
//replace github.com/nlpodyssey/safetensors => ../safetensors
