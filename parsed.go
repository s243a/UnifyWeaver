package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
)

func main() {

	// Read from stdin and process log_entry records with filtering
	regex1 := regexp.MustCompile(`([A-Z]+): (.+)`)
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ":")
		if len(parts) == 2 {
			field1 := parts[0]
			field2 := parts[1]
			matches := regex1.FindStringSubmatch(field2)
			if matches == nil || len(matches) != 3 {
				continue
			}

			cap1 := matches[1]
			cap2 := matches[2]
			result := field1 + ":" + cap1 + ":" + cap2
			if !seen[result] {
				seen[result] = true
				fmt.Println(result)
			}
		}
	}
}
