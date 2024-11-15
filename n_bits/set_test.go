// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import (
	"encoding/json"
	"strconv"
	"testing"
)

func TestBitSet(t *testing.T) {
	// Test Resize
	t.Run("0", func(t *testing.T) {
		l := 0
		b := &BitSet{}
		b.Resize(l)
		if b.Len != l {
			t.Errorf("expected length 100, got %d", b.Len)
		}
		if b.Effective() != 0 {
			t.Errorf("expected 0 effective bits, got %d", b.Effective())
		}
		bits := b.Expand()
		if len(bits) != b.Len {
			t.Errorf("expected %d bits in expanded slice, got %d", b.Len, len(bits))
		}
		d, err := b.MarshalJSON()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		var got BitSet
		if err := got.UnmarshalJSON(d); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if got.Len != b.Len {
			t.Errorf("expected length %d, got %d\nb:   %+v\ngot: %+v", b.Len, got.Len, b, &got)
		}
	})
	for l := 60; l < 64*3+2; l++ {
		t.Run(strconv.Itoa(l), func(t *testing.T) {
			b := &BitSet{}
			b.Resize(l)
			if b.Len != l {
				t.Errorf("expected length 100, got %d", b.Len)
			}

			b.Set(10)
			b.Set(50)
			if l > 99 {
				b.Set(99)
			}
			if !b.Get(10) {
				t.Errorf("expected bit 10 to be set")
			}
			if !b.Get(50) {
				t.Errorf("expected bit 50 to be set")
			}
			if l > 99 {
				if !b.Get(99) {
					t.Errorf("expected bit 99 to be set")
				}
			}
			if b.Get(0) {
				t.Errorf("expected bit 0 to be unset")
			}
			if b.Get(9) {
				t.Errorf("expected bit 9 to be unset")
			}
			if b.Get(11) {
				t.Errorf("expected bit 11 to be unset")
			}
			if l > 99 {
				if b.Get(98) {
					t.Errorf("expected bit 98 to be unset")
				}
			}

			if l == 100 && b.Effective() != 3 {
				t.Errorf("expected 3 effective bits, got %d", b.Effective())
			}

			bits := b.Expand()
			if len(bits) != b.Len {
				t.Errorf("expected %d bits in expanded slice, got %d", b.Len, len(bits))
			}
			for i := range bits {
				if i == 10 || i == 50 || (l > 99 && i == 99) {
					if !bits[i] {
						t.Errorf("expected bit %d to be set", i)
					}
				} else {
					if bits[i] {
						t.Errorf("expected bit %d to be unset", i)
					}
				}
			}

			d, err := b.MarshalJSON()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			var got BitSet
			if err := got.UnmarshalJSON(d); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if got.Len != b.Len {
				t.Errorf("expected length %d, got %d\nb:   %+v\ngot: %+v", b.Len, got.Len, b, &got)
			}
			for i := 0; i < b.Len; i++ {
				if b.Get(i) != got.Get(i) {
					t.Errorf("bit %d mismatch", i)
				}
			}
		})
	}
}

func TestCountSet(t *testing.T) {
	c := CountSet{Counts: make([]uint8, 5)}
	c.Resize(10)
	if len(c.Counts) != 10 {
		t.Errorf("Expected length 10, got %d", len(c.Counts))
	}
	c.Add(0)
	c.Add(0)
	c.Add(0)
	if c.Counts[0] != 3 {
		t.Errorf("Expected count 3, got %d", c.Counts[0])
	}
	for range 256 {
		c.Add(0)
	}
	if c.Counts[0] != 255 {
		t.Errorf("Expected count 255, got %d", c.Counts[0])
	}
	if c.Get(1) != 0 {
		t.Errorf("Expected 0, got %d", c.Get(1))
	}
	c.Counts = []uint8{1, 0, 3, 0, 0}
	if c.Effective() != 2 {
		t.Errorf("Expected 2 effective items, got %d", c.Effective())
	}

	c = CountSet{}
	b, err := json.Marshal(&c)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	got := CountSet{}
	if err = json.Unmarshal(b, &got); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(got.Counts) != 0 {
		t.Errorf("Unexpected deserialized value: %v", got.Counts)
	}

	c = CountSet{Counts: []uint8{1, 2, 3}}
	b, err = json.Marshal(&c)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	got = CountSet{}
	if err := json.Unmarshal(b, &got); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(got.Counts) != 3 || got.Counts[0] != 1 || got.Counts[1] != 2 || got.Counts[2] != 3 {
		t.Errorf("Unexpected deserialized value: %v", got.Counts)
	}
}
