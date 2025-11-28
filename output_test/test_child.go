package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {

	// Read from stdin and process parent records
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)
	
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ":")
		if len(parts) == 2 {
			field1 := parts[0]
			field2 := parts[1]
			result := field2 + ":" + field1
			if !seen[result] {
				seen[result] = true
				fmt.Println(result)
			}
		}
	}
}
