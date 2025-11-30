package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
)

func main() {

	// Read from stdin and process event records with regex filtering
	regex2 := regexp.MustCompile(`ERROR|WARNING|CRITICAL`)
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
					field1 := line
			if !regex2.MatchString(line) {
				continue
			}
		result := field1
		if !seen[result] {
			seen[result] = true
			fmt.Println(result)
		}
	}
}
