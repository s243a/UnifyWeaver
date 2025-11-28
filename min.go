package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

func main() {

	scanner := bufio.NewScanner(os.Stdin)
	var min float64
	first := true
	
	for scanner.Scan() {
		line := scanner.Text()
		val, err := strconv.ParseFloat(line, 64)
		if err == nil {
			if first || val < min {
				min = val
				first = false
			}
		}
	}
	
	if !first {
		fmt.Println(min)
	}
}
