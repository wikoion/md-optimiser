package main

import (
	"fmt"
	"regexp"
)

func main() {
	mds := []MachineDeployment{
		{"md-a-c7i", 32, 64, 10},
		{"md-b-general", 16, 64, 8},
		{"md-c-c7i", 64, 128, 5},
		{"md-d-burst", 32, 128, 6},
		{"md-e-c7i", 48, 96, 6},
		{"md-f-general", 24, 64, 4},
		{"md-g-c7i", 64, 128, 5},
		{"md-h-mixed", 48, 96, 4},
		{"md-i-c7i", 64, 192, 5},
		{"md-j-balanced", 32, 64, 6},
		{"md-k-c7i", 96, 128, 2},
		{"md-l-lowcost", 16, 64, 6},
		{"md-m-general", 24, 96, 4},
		{"md-n-c7i", 64, 128, 3},
		{"md-o-balanced", 48, 96, 3},
		{"md-o-m6id", 48, 96, 3},
	}

	pods := []Pod{
		{
			CPU:    5,
			Memory: 11,
			Labels: map[string]string{
				"workload-type": "nvme",
			},
		},
	}
	for i := 0; i < 50; i++ {
		pods = append(pods, Pod{
			CPU:    float64(2 + (i % 4)),        // 2–5 cores
			Memory: float64(2 + ((i * 3) % 10)), // 2–11 GB
		})
	}

	plugins := []ScoringPlugin{
		&FewestNodesPlugin{weight: 0.5},
		&LeastWastePlugin{weight: 0.5},
		&RegexMatchPlugin{weight: 0.9, pattern: regexp.MustCompile(`.*-c7i$`)},
	}

	assignments, nodesUsed := Optimise(mds, pods, plugins)

	fmt.Println("\nAssignments:")
	for i, j := range assignments {
		fmt.Printf("Pod %d → %s\n", i, mds[j].Name)
	}

	fmt.Println("\nNodes used:")
	for i, n := range nodesUsed {
		if n > 0 {
			fmt.Printf("%s → %d nodes\n", mds[i].Name, n)
		}
	}
}
