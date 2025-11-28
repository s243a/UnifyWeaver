package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
)

func main() {

	// Read from stdin and process with regex pattern matching

	pattern := regexp.MustCompile(`([0-9-]+) ([0-9:]+) ([A-Z]+)`)
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
		matches := pattern.FindStringSubmatch(line)
		if matches != nil {
			cap2 := matches[1]
			cap3 := matches[2]
			cap4 := matches[3]
			result := line + ":" + cap2 + ":" + cap3 + ":" + cap4
			if !seen[result] {
				seen[result] = true
				fmt.Println(result)
			}
		}
	}
}
