package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"strconv"
)

func main() {

	// Read from stdin and process person records with filtering

	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ":")
		if len(parts) == 2 {
			field1 := parts[0]
			field2 := parts[1]

			int2, err := strconv.Atoi(field2)
			if err != nil {
				continue
			}
			if !(int2 > 18) {
				continue
			}
			result := field1 + ":" + field2
			if !seen[result] {
				seen[result] = true
				fmt.Println(result)
			}
		}
	}
}
