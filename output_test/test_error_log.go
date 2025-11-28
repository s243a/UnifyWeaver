package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
)

func main() {

	// Read from stdin and process log records with regex filtering
	regex1 := regexp.MustCompile(`ERROR`)
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
					field1 := line
			if !regex1.MatchString(field1) {
				continue
			}
		result := field1
		if !seen[result] {
			seen[result] = true
			fmt.Println(result)
		}
	}
}
