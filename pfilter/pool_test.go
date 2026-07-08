package pfilter

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

// fakeCtxs returns n distinct fake pf_ctx handles. The pool never
// dereferences them, so any non-nil pointers work.
func fakeCtxs(n int) []unsafe.Pointer {
	backing := make([]byte, n)
	ctxs := make([]unsafe.Pointer, n)
	for i := range ctxs {
		ctxs[i] = unsafe.Pointer(&backing[i])
	}
	return ctxs
}

func TestCtxPoolBoundsConcurrency(t *testing.T) {
	const size, callers = 3, 20
	pool := newCtxPool(fakeCtxs(size))

	var inFlight, maxInFlight atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, ok := pool.acquire()
			if !ok {
				t.Error("acquire failed on an open pool")
				return
			}
			cur := inFlight.Add(1)
			for {
				prev := maxInFlight.Load()
				if cur <= prev || maxInFlight.CompareAndSwap(prev, cur) {
					break
				}
			}
			inFlight.Add(-1)
			pool.release(ctx)
		}()
	}
	wg.Wait()

	if got := maxInFlight.Load(); got > size {
		t.Errorf("max in-flight holders = %d, want <= pool size %d", got, size)
	}
}

func TestCtxPoolCloseAll(t *testing.T) {
	ctxs := fakeCtxs(2)
	pool := newCtxPool(ctxs)

	// Hold one context across closeAll to prove it waits for in-flight work.
	held, _ := pool.acquire()
	released := make(chan struct{})
	go func() {
		<-released
		pool.release(held)
	}()

	var freed []unsafe.Pointer
	var mu sync.Mutex
	done := make(chan struct{})
	go func() {
		pool.closeAll(func(ctx unsafe.Pointer) {
			mu.Lock()
			freed = append(freed, ctx)
			mu.Unlock()
		})
		close(done)
	}()

	// closeAll drains the idle context first, then must block on the held
	// one. Wait for that first free so the subsequent not-done check is
	// meaningful rather than racing closeAll's startup.
	for {
		mu.Lock()
		n := len(freed)
		mu.Unlock()
		if n == 1 {
			break
		}
		runtime.Gosched()
	}
	select {
	case <-done:
		t.Fatal("closeAll returned while a context was still held")
	default:
	}
	close(released)
	<-done

	if len(freed) != len(ctxs) {
		t.Fatalf("freed %d contexts, want %d", len(freed), len(ctxs))
	}
	seen := map[unsafe.Pointer]bool{}
	for _, ctx := range freed {
		seen[ctx] = true
	}
	for _, ctx := range ctxs {
		if !seen[ctx] {
			t.Errorf("context %p was never freed", ctx)
		}
	}

	if _, ok := pool.acquire(); ok {
		t.Error("acquire after closeAll must report a closed pool")
	}
}
