package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {

	facts := map[string]bool{
		"john:25": true,
		"jane:30": true,
		"bob:28": true,
	}

	for key := range facts {
		fmt.Println(key)
	}
}
