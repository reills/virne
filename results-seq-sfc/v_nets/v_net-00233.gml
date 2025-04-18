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
  id 233
  arrival_time 4221.975621776708
  lifetime 196.15314664112424
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 27
    rom 15
  ]
  node [
    id 1
    label "1"
    cpu 1
    gpu 38
    rom 33
  ]
  node [
    id 2
    label "2"
    cpu 12
    gpu 47
    rom 42
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 50
    rom 38
  ]
  node [
    id 4
    label "4"
    cpu 6
    gpu 3
    rom 23
  ]
  node [
    id 5
    label "5"
    cpu 35
    gpu 29
    rom 41
  ]
  node [
    id 6
    label "6"
    cpu 20
    gpu 39
    rom 37
  ]
  node [
    id 7
    label "7"
    cpu 0
    gpu 20
    rom 14
  ]
  node [
    id 8
    label "8"
    cpu 44
    gpu 21
    rom 40
  ]
  node [
    id 9
    label "9"
    cpu 37
    gpu 23
    rom 17
  ]
  node [
    id 10
    label "10"
    cpu 42
    gpu 49
    rom 9
  ]
  node [
    id 11
    label "11"
    cpu 12
    gpu 42
    rom 41
  ]
  edge [
    source 0
    target 1
    bw 43
  ]
  edge [
    source 1
    target 2
    bw 40
  ]
  edge [
    source 2
    target 3
    bw 29
  ]
  edge [
    source 3
    target 4
    bw 12
  ]
  edge [
    source 4
    target 5
    bw 8
  ]
  edge [
    source 5
    target 6
    bw 32
  ]
  edge [
    source 6
    target 7
    bw 7
  ]
  edge [
    source 7
    target 8
    bw 49
  ]
  edge [
    source 8
    target 9
    bw 14
  ]
  edge [
    source 9
    target 10
    bw 32
  ]
  edge [
    source 10
    target 11
    bw 48
  ]
]
