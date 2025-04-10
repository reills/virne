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
  id 1372
  arrival_time 29165.024738300115
  lifetime 572.644886037832
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 16
    gpu 19
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 9
    gpu 36
    rom 8
  ]
  node [
    id 2
    label "2"
    cpu 0
    gpu 45
    rom 27
  ]
  node [
    id 3
    label "3"
    cpu 20
    gpu 12
    rom 4
  ]
  node [
    id 4
    label "4"
    cpu 15
    gpu 37
    rom 42
  ]
  node [
    id 5
    label "5"
    cpu 18
    gpu 20
    rom 38
  ]
  node [
    id 6
    label "6"
    cpu 2
    gpu 25
    rom 32
  ]
  node [
    id 7
    label "7"
    cpu 22
    gpu 40
    rom 17
  ]
  node [
    id 8
    label "8"
    cpu 7
    gpu 35
    rom 37
  ]
  node [
    id 9
    label "9"
    cpu 25
    gpu 14
    rom 19
  ]
  node [
    id 10
    label "10"
    cpu 28
    gpu 3
    rom 39
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 35
  ]
  edge [
    source 2
    target 3
    bw 37
  ]
  edge [
    source 3
    target 4
    bw 9
  ]
  edge [
    source 4
    target 5
    bw 46
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 25
  ]
  edge [
    source 7
    target 8
    bw 31
  ]
  edge [
    source 8
    target 9
    bw 47
  ]
  edge [
    source 9
    target 10
    bw 7
  ]
]
