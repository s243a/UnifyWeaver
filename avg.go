package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

func main() {

	scanner := bufio.NewScanner(os.Stdin)
	sum := 0.0
	count := 0
	
	for scanner.Scan() {
		line := scanner.Text()
		val, err := strconv.ParseFloat(line, 64)
		if err == nil {
			sum += val
			count++
		}
	}
	
	if count > 0 {
		fmt.Println(sum / float64(count))
	}
}
