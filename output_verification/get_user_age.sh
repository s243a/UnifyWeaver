#!/bin/bash
# get_user_age - streaming pipeline with uniqueness (sort -u)



get_user_age() {
    users_stream | sort -u
}

# Stream function for use in pipelines
get_user_age_stream() {
    get_user_age
}