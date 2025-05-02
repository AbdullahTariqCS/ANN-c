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

    for(int i = 0; i < *count; i++)
    {
        entry = readdir(dir);
        (*names)[i] = atoi(entry->d_name); 
    }

    closedir(dir);
}
