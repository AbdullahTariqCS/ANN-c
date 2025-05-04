#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_dir(int **names, int *count, const char *dir_name)
{
    DIR *dir;
    struct dirent *entry;

    int local_count = 0;
    dir = opendir(dir_name);
    if (!dir) {
        fprintf(stderr, "Cannot open directory: %s\n", dir_name);
        exit(EXIT_FAILURE);
    }

    while ((entry = readdir(dir)) != NULL) {
        // if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) 
        //     continue;
        local_count++;
    }
    closedir(dir);

    if(*count = -1 || *count > local_count)
        *count = local_count;

    *names = malloc(*count * sizeof(int));
    if (!*names) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }

    dir = opendir(dir_name);
    if (!dir) {
        fprintf(stderr, "Cannot reopen directory: %s\n", dir_name);
        exit(EXIT_FAILURE);
    }

    int idx = 0;
    while ((entry = readdir(dir)) != NULL) {
        // if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) 
        //     continue;
        if (idx >= *count) break;
        (*names)[idx] = atoi(entry->d_name);
        idx++;
    }

    closedir(dir);
}
