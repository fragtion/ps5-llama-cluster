// rpc-cache-check [-n] DIR...: verify llama.cpp RPC weight-cache files and delete damaged ones.
// ggml-rpc-server names each cache file after the FNV-1a 64 hash of its contents but never checks
// it when loading, so a file damaged by a failing drive or cut short by a power loss would load as
// bad weights. A deleted file is simply sent over the network again at the next pool start.
// -n only reports. Missing directories are skipped. Build: gcc -O2 -o rpc-cache-check rpc-cache-check.c
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int check_file(int dfd, const char *name, uint64_t *bytes) {
    static unsigned char buf[4 << 20];
    int fd = openat(dfd, name, O_RDONLY);
    if (fd < 0) return -1;
    uint64_t h = 0xcbf29ce484222325ULL;
    off_t pos = 0;
    ssize_t got;
    while ((got = read(fd, buf, sizeof buf)) > 0) {
        for (ssize_t i = 0; i < got; i++) { h ^= buf[i]; h *= 0x100000001b3ULL; }
        posix_fadvise(fd, pos, got, POSIX_FADV_DONTNEED);   // don't push the model out of RAM
        pos += got;
    }
    close(fd);
    if (got < 0) return 0;   // unreadable: treat as damaged
    *bytes += pos;
    char want[17];
    snprintf(want, sizeof want, "%016" PRIx64, h);
    return strcmp(want, name) == 0;
}

int main(int argc, char **argv) {
    int dry = argc > 1 && strcmp(argv[1], "-n") == 0;
    int bad_total = 0;
    for (int a = 1 + dry; a < argc; a++) {
        DIR *d = opendir(argv[a]);
        if (!d) continue;
        int ok = 0, bad = 0;
        uint64_t bytes = 0;
        struct dirent *e;
        while ((e = readdir(d))) {
            if (strlen(e->d_name) != 16 || strspn(e->d_name, "0123456789abcdef") != 16) continue;
            int r = check_file(dirfd(d), e->d_name, &bytes);
            if (r == 1) { ok++; continue; }
            bad++;
            printf("%s/%s: damaged%s\n", argv[a], e->d_name, dry ? "" : ", deleted");
            if (!dry) unlinkat(dirfd(d), e->d_name, 0);
        }
        closedir(d);
        printf("%s: %d files ok (%.2f GB), %d damaged\n", argv[a], ok, bytes / 1e9, bad);
        bad_total += bad;
    }
    return bad_total ? 1 : 0;
}
