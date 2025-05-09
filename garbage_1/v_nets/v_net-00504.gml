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
  id 504
  arrival_time 9483.959909215311
  lifetime 793.4284706007685
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 36
    gpu 36
    rom 39
  ]
  node [
    id 1
    label "1"
    cpu 9
    gpu 48
    rom 25
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 16
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 22
    gpu 14
    rom 44
  ]
  node [
    id 4
    label "4"
    cpu 3
    gpu 14
    rom 25
  ]
  node [
    id 5
    label "5"
    cpu 48
    gpu 48
    rom 36
  ]
  node [
    id 6
    label "6"
    cpu 44
    gpu 46
    rom 33
  ]
  node [
    id 7
    label "7"
    cpu 37
    gpu 18
    rom 21
  ]
  node [
    id 8
    label "8"
    cpu 38
    gpu 28
    rom 28
  ]
  node [
    id 9
    label "9"
    cpu 41
    gpu 43
    rom 27
  ]
  edge [
    source 0
    target 1
    bw 46
  ]
  edge [
    source 1
    target 2
    bw 45
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
  edge [
    source 3
    target 4
    bw 32
  ]
  edge [
    source 4
    target 5
    bw 39
  ]
  edge [
    source 5
    target 6
    bw 45
  ]
  edge [
    source 6
    target 7
    bw 47
  ]
  edge [
    source 7
    target 8
    bw 49
  ]
  edge [
    source 8
    target 9
    bw 22
  ]
]
