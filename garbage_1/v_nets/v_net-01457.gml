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
  id 1457
  arrival_time 30621.239292415627
  lifetime 7642.26511042113
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 32
    gpu 15
    rom 26
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 5
    rom 13
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 27
    rom 21
  ]
  node [
    id 3
    label "3"
    cpu 44
    gpu 41
    rom 23
  ]
  node [
    id 4
    label "4"
    cpu 19
    gpu 31
    rom 19
  ]
  node [
    id 5
    label "5"
    cpu 49
    gpu 50
    rom 31
  ]
  node [
    id 6
    label "6"
    cpu 31
    gpu 45
    rom 50
  ]
  node [
    id 7
    label "7"
    cpu 0
    gpu 47
    rom 29
  ]
  node [
    id 8
    label "8"
    cpu 39
    gpu 38
    rom 17
  ]
  node [
    id 9
    label "9"
    cpu 15
    gpu 17
    rom 26
  ]
  node [
    id 10
    label "10"
    cpu 33
    gpu 30
    rom 45
  ]
  node [
    id 11
    label "11"
    cpu 41
    gpu 26
    rom 43
  ]
  edge [
    source 0
    target 1
    bw 37
  ]
  edge [
    source 1
    target 2
    bw 8
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
  edge [
    source 3
    target 4
    bw 1
  ]
  edge [
    source 4
    target 5
    bw 15
  ]
  edge [
    source 5
    target 6
    bw 33
  ]
  edge [
    source 6
    target 7
    bw 9
  ]
  edge [
    source 7
    target 8
    bw 22
  ]
  edge [
    source 8
    target 9
    bw 50
  ]
  edge [
    source 9
    target 10
    bw 13
  ]
  edge [
    source 10
    target 11
    bw 14
  ]
]
