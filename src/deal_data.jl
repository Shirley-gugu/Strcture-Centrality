function modify_file(input_file::AbstractString, output_file::AbstractString)
    open(output_file, "w") do out
        for line in eachline(input_file)
            parts = split(line)

            # 检查行是否有足够的数字
            if length(parts) >= 3
                num = parse(Float32, parts[3])  # 解析第三个数字
                if rand() < 0.15  # 以15%的概率将其修改为-1
                    num = -1
                else
                    num = 1
                end
                parts[3] = string(num)  # 修改第三个数字

                modified_line = join(parts, " ")  # 重新组合每行
                println(out, modified_line)  # 写入新文件
            else
                println(out, line)  # 如果行没有足够的数字，则写入原始行
            end
        end
    end
end

# 示例用法

# modify_file("data/hamster-household.txt", "data/hamster-household1.txt")

function delete_two_line(filename)
    lines=readlines(filename)
    lines=lines[3:end]
    write(filename, join(lines, "\n"))
end

function add_random_number(filename::AbstractString)
    lines = readlines(filename)
    for i in 1:length(lines)
        if rand() < 0.15
            lines[i] = string(lines[i], " -1")
        else
            lines[i] = string(lines[i], " 1")
        end
    end

    # 将修改后的内容写回文件
    write(filename, join(lines, "\n"))
end

function delete_last_one(filename::AbstractString)
    lines=readlines(filename)
    for i in 1:length(lines)
        parts=split(lines[i])
        if length(parts)>=3
            parts=parts[1:end-1]
            modified_line=join(parts, " ")
            lines[i]=modified_line
        end
    end
    write(filename, join(lines, "\n"))
end

function delete_last_two(filename::AbstractString)
    lines=readlines(filename)
    for i in 1:length(lines)
        parts=split(lines[i])
        if length(parts)>=3
            parts=parts[1:end-2]
            modified_line=join(parts, " ")
            lines[i]=modified_line
        end
    end
    write(filename, join(lines, "\n"))
end

function format(filename::AbstractString)
    lines=readlines(filename)
    for i in eachindex(lines)
        parts=split(lines[i])
        modified_line=join(parts, " ")
        lines[i]=modified_line
    end
    write(filename, join(lines, "\n"))
end

function add_random_weight(filename::AbstractString)
    lines=readlines(filename)
    for i in eachindex(lines)
        parts=split(lines[i])
        if rand()<0.15
            r=rand(-10:-1)
        else
            r=rand(1:10)
        end
        if length(parts)>=3
            parts[3]=string(r)
            parts=parts[1:3]
            modified_line=join(parts, " ")
            lines[i]=modified_line
        else 
            lines[i]=string(lines[i], " ", r)
        end
    end
    write(filename, join(lines, "\n"))
end

# delete_two_line("data/dbpedia-link.txt")
# add_random_number("data/producer.txt")
# delete_last_two("data/epinions.txt")
# delete_last_one("data/epinions.txt")
# format("data/wikipedialinks.txt")

# add_random_weight("data/input/test.txt")
add_random_number("data/input/webbase.txt")
