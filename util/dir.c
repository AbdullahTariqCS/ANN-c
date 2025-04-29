#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_dir(char ***names, int *count, const char *dir_name)
{
    DIR *dir;
    struct dirent *entry;

    if (!count)
    {
        *count = 0;
    
        dir = opendir(dir_name);
        if (!dir) {
            fprintf(stderr, "Cannot open directory: %s\n", dir_name);
            exit(EXIT_FAILURE);
        }
    
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG)  // regular file
                (*count)++;
        }
        closedir(dir);
    }

    *names = malloc(*count * sizeof(char *));
    if (!*names) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }

    dir = opendir(dir_name);
    if (!dir) {
        fprintf(stderr, "Cannot reopen directory: %s\n", dir_name);
        exit(EXIT_FAILURE);
    }

    int i = 0;
    for(int i = 0; i < *count; i++)
    {
        entry = readdir(dir);
        if (entry->d_type == DT_REG) {
            (*names)[i] = strdup(entry->d_name); // copy name into new memory
            if (!(*names)[i]) {
                perror("strdup failed");
                exit(EXIT_FAILURE);
            }
        }
    }

    closedir(dir);
}
