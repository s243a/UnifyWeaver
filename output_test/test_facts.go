package main

import (
	"fmt"
)

func main() {

	facts := map[string]bool{
		"alice:25": true,
		"bob:30": true,
		"charlie:28": true,
	}

	for key := range facts {
		fmt.Println(key)
	}
}
