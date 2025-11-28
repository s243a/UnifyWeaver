package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

func main() {

	scanner := bufio.NewScanner(os.Stdin)
	var max float64
	first := true
	
	for scanner.Scan() {
		line := scanner.Text()
		val, err := strconv.ParseFloat(line, 64)
		if err == nil {
			if first || val > max {
				max = val
				first = false
			}
		}
	}
	
	if !first {
		fmt.Println(max)
	}
}
