graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 242
  arrival_time 4540.610242647645
  lifetime 1210.7952766172648
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 42
    gpu 36
    rom 42
  ]
  node [
    id 1
    label "1"
    cpu 21
    gpu 11
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 14
    gpu 9
    rom 37
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 47
    rom 50
  ]
  node [
    id 4
    label "4"
    cpu 16
    gpu 5
    rom 35
  ]
  node [
    id 5
    label "5"
    cpu 49
    gpu 47
    rom 26
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 40
    rom 27
  ]
  node [
    id 7
    label "7"
    cpu 39
    gpu 19
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 6
    gpu 9
    rom 50
  ]
  node [
    id 9
    label "9"
    cpu 29
    gpu 12
    rom 6
  ]
  edge [
    source 0
    target 1
    bw 29
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 17
  ]
  edge [
    source 3
    target 4
    bw 40
  ]
  edge [
    source 4
    target 5
    bw 19
  ]
  edge [
    source 5
    target 6
    bw 13
  ]
  edge [
    source 6
    target 7
    bw 47
  ]
  edge [
    source 7
    target 8
    bw 20
  ]
  edge [
    source 8
    target 9
    bw 18
  ]
]
