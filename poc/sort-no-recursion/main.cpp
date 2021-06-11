#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main(int argc, char** argv) {

  if (argc == 1) {
    return -1;
  }

  int n = std::stoi(argv[1]);

  std::cout << "n = " << n << std::endl;

  std::vector<int> my_array(n);

  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count(); // std::stoi(argv[1]);
  std::default_random_engine generator(seed1);
  std::uniform_int_distribution<int> distribution(0,10);

  for (auto &e : my_array) {
    e = distribution(generator);
  }  

  std::cout << "my_array values = ";
  for (auto &e : my_array) {
    std::cout << e << " ";
  }
  std::cout << std::endl;

  std::vector<int> sorted_array(n);
  int stride = 1;
  while (stride < n) {
    for (int offset = 0; offset < n; offset += stride*2) {
      int p = offset; 
      int q = p + stride;
      int r = q + stride < n ? q + stride : n;
      int i = p;
      int j = q;
      for (int k = p; k < r; k++) {
        if (i < q && j < r) {
	  if (my_array[i] < my_array[j]) {
	    sorted_array[k] = my_array[i];
	    i++;
	  }
	  else {
	    sorted_array[k] = my_array[j];
	    j++;
	  }
        }
	else {
          if (i < q) {
	    sorted_array[k] = my_array[i];
	    i++;
          } else {
	    sorted_array[k] = my_array[j];
	    j++;
          }
        }
      }
    }
    my_array = sorted_array;
    stride *= 2;
  }

  std::cout << "sorted my_array values = ";
  for (auto &e : my_array) {
    std::cout << e << " ";
  }
  std::cout << std::endl;

  return 0;
}
