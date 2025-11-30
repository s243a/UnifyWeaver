package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
)

func main() {

	// Read from stdin and process with regex pattern matching

	pattern := regexp.MustCompile(`([0-9-]+ [0-9:]+)`)
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
		matches := pattern.FindStringSubmatch(line)
		if matches != nil {
			cap2 := matches[1]
			result := line + ":" + cap2
			if !seen[result] {
				seen[result] = true
				fmt.Println(result)
			}
		}
	}
}
