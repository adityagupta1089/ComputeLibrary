all_cpus=("0-3" "4-7" "0-7")
all_graphs=("graph_alexnet" "graph_googlenet" "graph_mobilenet" "graph_resnet50"  "graph_squeezenet")
for cpus in $all_cpus; do
	cpu1=$(echo $cpus | cut -d'-' -f1)
	cpu2=$(echo $cpus | cut -d'-' -f2)
	n=$(($cpu2 - $cpu1 + 1))
	maxn=$((10 * $n))
	for ((p = $n; p <= $maxn; p += $n)); do
		for graph in $all_graphs; do
			command="taskset -c $cpus ./$graph --threads=$p"
			out=$($command)
			if $(echo $out | grep -q "Test passed"); then 
				result=$(echo $out | grep -o "Test.*")
				cost=$(echo $out | grep -oP "(?<=Cost: )[\d.]+";)
				echo -e "$graph\t\t: $cost\tsec\t[$command, $result]"
			else
				echo -e "$graph\t\t: 0\tsec\t[Test failed]"
			fi		
		done
	done
done

