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
	
	for scanner.Scan() {
		line := scanner.Text()
		val, err := strconv.ParseFloat(line, 64)
		if err == nil {
			sum += val
		}
	}
	
	fmt.Println(sum)
}
