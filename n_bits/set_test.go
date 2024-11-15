// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import "testing"

func TestBitSet(t *testing.T) {
	// Test Resize
	b := &BitSet{}
	b.Resize(100)
	if b.Len != 100 {
		t.Errorf("expected length 100, got %d", b.Len)
	}

	// Test Set and Get
	b.Set(10)
	b.Set(50)
	b.Set(99)
	if !b.Get(10) || !b.Get(50) || !b.Get(99) {
		t.Errorf("expected bits 10, 50, 99 to be set")
	}
	if b.Get(0) || b.Get(11) || b.Get(51) {
		t.Errorf("expected bits 0, 11, 51 to be unset")
	}

	// Test Effective
	if b.Effective() != 3 {
		t.Errorf("expected 3 effective bits, got %d", b.Effective())
	}

	// Test Expand
	bits := b.Expand()
	if len(bits) != b.Len {
		t.Errorf("expected %d bits in expanded slice, got %d", b.Len, len(bits))
	}
	for i := range bits {
		if i == 10 || i == 50 || i == 99 {
			if !bits[i] {
				t.Errorf("expected bit %d to be set", i)
			}
		} else {
			if bits[i] {
				t.Errorf("expected bit %d to be unset", i)
			}
		}
	}

	// Test JSON Marshal/Unmarshal
	jsonData, err := b.MarshalJSON()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	var b2 BitSet
	if err := b2.UnmarshalJSON(jsonData); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if b2.Len != b.Len {
		t.Errorf("expected length %d, got %d", b.Len, b2.Len)
	}
	for i := 0; i < b.Len; i++ {
		if b.Get(i) != b2.Get(i) {
			t.Errorf("bit %d mismatch", i)
		}
	}
}
