#include <iostream>
#include "ml\models\linear_regression.h"

int main()
{
    ml::LinearRegression lr = ml::LinearRegression(0.01, 1000);
    lr.train({{1, 1}, {2, 2}, {3, 3}, {4, 4}}, {1, 2, 3, 4});
    std::cout << lr.predict({5, 5}) << std::endl;
    std::cout << lr.predict({6, 6}) << std::endl;
    return 0;
}