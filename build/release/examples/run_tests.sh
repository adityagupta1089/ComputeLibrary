all_cpus=("0-7")
all_graphs=("graph_alexnet" "graph_googlenet" "graph_mobilenet" "graph_resnet50"  "graph_squeezenet")
for cpus in ${all_cpus[@]}; do
	cpu1=$(echo $cpus | cut -d'-' -f1)
	cpu2=$(echo $cpus | cut -d'-' -f2)
	n=$(($cpu2 - $cpu1 + 1))
	for graph in ${all_graphs[@]}; do
		command="taskset -c $cpus ./$graph --threads=$n"
		out=$(eval "timeout 120 $command")
		if $(echo $out | grep -q "Test passed"); then 
			result=$(echo $out | grep -o "Test.*")3
			cost=$(echo $out | grep -oP "(?<=Cost: )[\d.]+";)
			echo -e "$graph\t\t\t: $cost\tsec\t[$command, $result]"
		else
			echo -e "$graph\t\t\t: 0\tsec\t[Test failed]"
		fi		
	done
done
