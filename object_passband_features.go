package main

import (
	"strconv"

	"github.com/MaxHalford/tuna"
)

// BrightFaintRatio computes feature A of section 2.2 of
// "Kim, Dae-Won, and Coryn 2015".
type BrightFaintRatio struct {
	n   float64
	mu  float64 // Running mean
	bss float64 // Running sum of squares of values brighter than the mean
	fss float64 // Running sum of squares of values fainter than the mean
}

// Update BrightFaintRatio given a Row.
func (bfr *BrightFaintRatio) Update(row tuna.Row) error {
	x, _ := strconv.ParseFloat(row["flux"], 64)
	bfr.n++
	// Compute the current mean
	mu := bfr.mu + (x-bfr.mu)/bfr.n
	// Update the appropriate sum of squares and the mean
	if x > mu {
		bfr.bss += (x - bfr.mu) * (x - mu)
	} else {
		bfr.fss += (x - bfr.mu) * (x - mu)
	}
	bfr.mu = mu
	return nil
}

// Collect returns the current value.
func (bfr BrightFaintRatio) Collect() <-chan tuna.Row {
	c := make(chan tuna.Row)
	go func() {
		c <- tuna.Row{"flux_bfr": strconv.FormatFloat(bfr.bss/(bfr.fss+1), 'f', -1, 64)}
		close(c)
	}()
	return c
}

// Size is 1.
func (bfr BrightFaintRatio) Size() uint { return 1 }

func str2Float(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}

func calcObjectFeatures() {
	// Instantiate a CSV stream to read the train and test rows one by one
	train, err := tuna.NewCSVStreamFromPath("data/training_set.csv")
	if err != nil {
		panic(err)
	}
	test, err := tuna.NewCSVStreamFromPath("data/test_set.csv")
	if err != nil {
		panic(err)
	}
	stream := tuna.ZipStreams(train, test)

	// Define a Sink to output the results
	sink, err := tuna.NewCSVSinkFromPath("data/features/flux_stats2.csv")
	if err != nil {
		panic(err)
	}

	// Define some feature extractors
	extractor := tuna.NewSequentialGroupBy(
		"object_id",
		func() tuna.Extractor {
			return tuna.NewUnion(
				tuna.NewNUnique("passband"),
				tuna.NewCount(),
			)
		},
		sink,
	)

	// Run the feature extractors over the stream
	if err = tuna.Run(stream, extractor, nil, 1e5); err != nil {
		panic(err)
	}
}

func calcObjectPassbandFeatures() {

	// Instantiate a CSV stream to read the train and test rows one by one
	train, err := tuna.NewCSVStreamFromPath("data/training_set.csv")
	if err != nil {
		panic(err)
	}
	test, err := tuna.NewCSVStreamFromPath("data/test_set.csv")
	if err != nil {
		panic(err)
	}
	stream := tuna.ZipStreams(train, test)

	// Define a Sink to output the results
	sink, err := tuna.NewCSVSinkFromPath("data/features/flux_stats.csv")
	if err != nil {
		panic(err)
	}

	// Define some feature extractors
	extractor := tuna.NewSequentialGroupBy(
		"object_id",
		func() tuna.Extractor {
			return tuna.NewGroupBy(
				"passband",
				func() tuna.Extractor {
					return tuna.NewUnion(
						// Bright/faint ratio by (object_id, passband)
						&BrightFaintRatio{},
						// Flux mean by (object_id, passband)
						tuna.NewMean("flux"),
						// Flux min by (object_id, passband)
						tuna.NewMin("flux"),
						// Flux max by (object_id, passband)
						tuna.NewMax("flux"),
						// Flux PTP by (object_id, passband)
						tuna.NewPTP("flux"),
						// Flux skew by (object_id, passband)
						tuna.NewSkew("flux"),
						// Flux kurtosis by (object_id, passband)
						tuna.NewKurtosis("flux"),
						// Flux quantiles by (object_id, passband)
						tuna.NewQuantile("flux", 0.01, []float64{0.25, 0.5, 0.75}),
						// Flux error mean by (object_id, passband)
						tuna.NewMean("flux_err"),
						// Flux error min by (object_id, passband)
						tuna.NewMin("flux_err"),
						// Flux error max by (object_id, passband)
						tuna.NewMax("flux_err"),
						// Flux error PTP by (object_id, passband)
						tuna.NewPTP("flux_err"),
						// Flux error skew by (object_id, passband)
						tuna.NewSkew("flux_err"),
						// Flux kurtosis by (object_id, passband)
						tuna.NewKurtosis("flux_err"),
						// Flux differences by (object_id, passband)
						tuna.NewDiff(
							"flux",
							func(s string) tuna.Extractor {
								return tuna.NewUnion(
									// Bright/faint ratio by (object_id, passband)
									&BrightFaintRatio{},
									// Flux mean by (object_id, passband)
									tuna.NewMean("flux"),
									// Flux min by (object_id, passband)
									tuna.NewMin("flux"),
									// Flux max by (object_id, passband)
									tuna.NewMax("flux"),
									// Flux PTP by (object_id, passband)
									tuna.NewPTP("flux"),
									// Flux skew by (object_id, passband)
									tuna.NewSkew("flux"),
									// Flux kurtosis by (object_id, passband)
									tuna.NewKurtosis("flux"),
								)
							},
						),
					)
				},
			)
		},
		sink,
	)

	// Run the feature extractors over the stream
	if err = tuna.Run(stream, extractor, nil, 1e5); err != nil {
		panic(err)
	}
}

func main() {
	calcObjectFeatures()
	calcObjectPassbandFeatures()
}
