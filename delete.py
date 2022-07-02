import numpy as np
from typing import *


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1,-1]
        left = self.get_left_index(nums,target)
        right = self.get_right_index(nums, target)
        return [left,right]

    def get_left_index(self,nums: List[int], target: int):
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                right = mid-1
            if nums[mid] > target:
                right = mid-1
            if nums[mid] < target:
                left = mid+1
        if left < len(nums) and nums[left] == target:
            return left
        else:
            return -1

    def get_right_index(self,nums: List[int], target: int):
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                left = mid + 1
            if nums[mid] > target:
                right = mid - 1
            if nums[mid] < target:
                left = mid + 1
        if right >= 0 and nums[right] == target:
            return right
        else:
            return -1


if __name__ == '__main__':
    nums1 = [2,2]
    target1 = 3                      # [3, 4]
    s = Solution()
    ret1 = s.searchRange(nums1,target1)
    print(ret1)




