package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {

	scanner := bufio.NewScanner(os.Stdin)
	count := 0
	
	for scanner.Scan() {
		count++
	}
	
	fmt.Println(count)
}
