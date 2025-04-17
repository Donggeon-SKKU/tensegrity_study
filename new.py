# # Make Sigma_new
# import sympy as sp

# counter = 1
# def generate_Sigma_combination(elements,num_members,current=[],result=[]):
#     global counter
#     if len(current) == num_members:
#         counter += 1    
#         result.append(current)
#         print(counter)
#         return
    
#     for element in elements:
#         generate_Sigma_combination(elements,num_members,current + [element], result)

# num_member = 30
# l,g = sp.symbols('l g')
# elements = [l,g]
# result = []
# generate_Sigma_combination(elements, num_member,result=result)
# print(len(result))

import numpy as np
import cupy as cp  # GPU 계산을 위한 라이브러리
import sympy as sp

# 이진수 표현을 사용하여 조합 생성
def generate_combinations_gpu(num_members):
    # 총 조합 수 (2^num_members)
    total_combinations = 2**num_members
    
    # 배치 크기 설정 (메모리 한계에 따라 조정)
    batch_size = 1000000
    
    l, g = sp.symbols('l g')
    results = []
    
    # 배치 단위로 처리
    for start in range(0, total_combinations, batch_size):
        end = min(start + batch_size, total_combinations)
        print(f"처리 중: {start}/{total_combinations} ~ {end}/{total_combinations}")
        
        # GPU에서 이진 표현 생성
        indices = cp.arange(start, end, dtype=cp.int64)
        
        # 이진 표현을 CPU로 가져와서 심볼로 변환
        indices_cpu = cp.asnumpy(indices)
        
        # 각 숫자를 이진수로 변환하고 l, g 심볼로 매핑
        for idx in indices_cpu:
            binary = format(idx, f'0{num_members}b')
            combination = [l if bit == '0' else g for bit in binary]
            results.append(combination)
            
            # 진행 상황 출력 (10만개마다)
            if len(results) % 100000 == 0:
                print(f"생성된 조합 수: {len(results)}")
    
    return results

# 예시 사용 (작은 수로 테스트)
num_member = 10  # 우선 작은 수로 테스트
result = generate_combinations_gpu(num_member)
print(f"총 조합 수: {len(result)}")