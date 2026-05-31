#include <stdio.h>

int main(int argc, char **v) {
  int c;
  while (*v) {
    printf("arg.cu: %s\n", *v);
    v++;
  }
  while ((c = getchar()) != EOF) {
    putchar(c);
  }
  return 0;
}
