from collections import deque

def num_islands_bfs(m, n, matrix):
    num_islands = 0
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    queue = deque()

    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                queue.append((i,j))
                num_islands += 1

                while queue:
                    r, c = queue.popleft()
                    matrix[r][c] = '0'
                    for d in dirs:
                        next_r, next_c = r + d[0], c + d[1]
                        if 0 <= next_r < m and 0 <= next_c < n and matrix[next_r][next_c] == '1':
                            queue.append((next_r, next_c))

    return num_islands


def main():
    m, n = map(int, input("Enter dimensions: ").strip().split())

    print("Enter the matrix row by row:")
    matrix = []
    for _ in range(m):
        row = input().strip().split()
        matrix.append(row)

    result = num_islands_bfs(m, n, matrix)
    print(f"Number of islands: {result}")


if __name__ == "__main__":
    main()
