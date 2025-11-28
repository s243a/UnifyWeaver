package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {

	facts := map[string]bool{
		"alice:bob": true,
		"bob:charlie": true,
		"alice:dave": true,
	}

	for key := range facts {
		fmt.Println(key)
	}
}
