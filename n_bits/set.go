// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math/bits"
)

// Note: there's many many high efficiency bit sets but few with counts? I
// didn't search much yet. Here's a few path to look for. It's mostly important
// for the mantissa of both float32 and int32.
// - https://golang.org/x/tools/container/intsets
// - https://www.gonum.org/
// - https:/github.com/james-bowman/sparse
// - https://github.com/RoaringBitmap/roaring
// Investigate getting rid of the code below.

// BitSet is a bit set.
//
// It is designed to be densely stored in JSON.
type BitSet struct {
	Len  int
	Bits []uint64
}

func (b *BitSet) Resize(l int) {
	d := make([]uint64, (l+63)/64)
	// Backup the old data if any.
	copy(b.Bits, d)
	b.Len = l
	b.Bits = d
}

func (b *BitSet) Set(i int) {
	b.Bits[i/64] |= 1 << (i % 64)
}

func (b *BitSet) Get(i int) bool {
	return b.Bits[i/64]&(1<<(i%64)) != 0
}

func (b *BitSet) Expand() []bool {
	out := make([]bool, b.Len)
	// TODO: This is the slow version.
	for i := range b.Len {
		out[i] = b.Get(i)
	}
	return out
}

// Effective returns the number of non-zero items in the slice.
func (b *BitSet) Effective() int32 {
	o := 0
	for _, v := range b.Bits {
		o += bits.OnesCount64(v)
	}
	return int32(o)
}

// MarshalJSON implements json.Marshaler
//
// The first byte is the number of valid bits in the last uint64. If 0, it
// means 64.
func (b *BitSet) MarshalJSON() ([]byte, error) {
	var dst []byte
	if b.Len != 0 {
		d := make([]byte, 1, len(b.Bits)*8+1)
		d[0] = byte(b.Len % 64)
		var buf [8]byte
		for _, v := range b.Bits {
			binary.LittleEndian.PutUint64(buf[:], v)
			d = append(d, buf[:]...)
		}
		dst = make([]byte, base64.RawStdEncoding.EncodedLen(len(d)))
		base64.RawStdEncoding.Encode(dst, d)
	}
	return json.Marshal(string(dst))
}

// UnmarshalJSON implements json.Unmarshaler
func (b *BitSet) UnmarshalJSON(data []byte) error {
	s := ""
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	if len(s) == 0 {
		b.Len = 0
		b.Bits = nil
		return nil
	}
	d, err := base64.RawStdEncoding.DecodeString(s)
	if err != nil {
		return err
	}
	if len(d) == 0 {
		return errors.New("invalid BitSet base64 encoding")
	}
	last := d[0]
	if last > 63 {
		return errors.New("invalid BitSet encoding")
	}
	if last == 0 {
		last = 64
	}
	l := (len(d) + 6) / 8
	b.Bits = make([]uint64, l)
	for i := range b.Bits {
		// Note: Lots of unaligned reads.
		b.Bits[i] = binary.LittleEndian.Uint64(d[1+i*8 : 9+i*8])
	}
	b.Len = (l-1)*64 + int(last)
	return nil
}

// CountSet is a count set.
//
// It is designed to be densely stored in JSON.
//
// TODO: Handle overflows.
type CountSet struct {
	Counts []uint8
}

func (c *CountSet) Resize(l int) {
	d := make([]uint8, l)
	// Backup the old data if any.
	copy(c.Counts, d)
	c.Counts = d
}

func (c *CountSet) Add(i int) {
	if c.Counts[i] != 0xFF {
		c.Counts[i]++
	}
	// else handle overflow.
}

func (c *CountSet) Get(i int) uint8 {
	return c.Counts[i]
}

// Effective returns the number of non-zero items in the slice.
func (c *CountSet) Effective() int32 {
	o := 0
	for _, v := range c.Counts {
		if v != 0 {
			o += 1
		}
	}
	return int32(o)
}

// MarshalJSON implements json.Marshaler
func (c *CountSet) MarshalJSON() ([]byte, error) {
	var dst []byte
	if len(c.Counts) != 0 {
		dst = make([]byte, base64.RawStdEncoding.EncodedLen(len(c.Counts)))
		base64.RawStdEncoding.Encode(dst, c.Counts)
	}
	return json.Marshal(string(dst))
}

// UnmarshalJSON implements json.Unmarshaler
func (c *CountSet) UnmarshalJSON(data []byte) error {
	s := ""
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	if len(s) == 0 {
		c.Counts = nil
		return nil
	}
	d, err := base64.RawStdEncoding.DecodeString(s)
	if err != nil {
		return err
	}
	if len(d) == 0 {
		return errors.New("invalid BitSet base64 encoding")
	}
	c.Counts = d
	return nil
}
