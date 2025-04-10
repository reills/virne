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
  id 1756
  arrival_time 39134.965966211326
  lifetime 103.65186391723272
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 37
    gpu 4
    rom 28
  ]
  node [
    id 1
    label "1"
    cpu 10
    gpu 3
    rom 25
  ]
  node [
    id 2
    label "2"
    cpu 5
    gpu 21
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 50
    gpu 22
    rom 41
  ]
  node [
    id 4
    label "4"
    cpu 44
    gpu 1
    rom 8
  ]
  node [
    id 5
    label "5"
    cpu 25
    gpu 39
    rom 13
  ]
  node [
    id 6
    label "6"
    cpu 31
    gpu 27
    rom 47
  ]
  node [
    id 7
    label "7"
    cpu 13
    gpu 28
    rom 28
  ]
  node [
    id 8
    label "8"
    cpu 31
    gpu 47
    rom 14
  ]
  node [
    id 9
    label "9"
    cpu 23
    gpu 8
    rom 14
  ]
  node [
    id 10
    label "10"
    cpu 36
    gpu 49
    rom 48
  ]
  node [
    id 11
    label "11"
    cpu 10
    gpu 20
    rom 7
  ]
  node [
    id 12
    label "12"
    cpu 1
    gpu 24
    rom 15
  ]
  node [
    id 13
    label "13"
    cpu 36
    gpu 23
    rom 15
  ]
  node [
    id 14
    label "14"
    cpu 1
    gpu 14
    rom 19
  ]
  edge [
    source 0
    target 1
    bw 43
  ]
  edge [
    source 1
    target 2
    bw 43
  ]
  edge [
    source 2
    target 3
    bw 4
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 3
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 25
  ]
  edge [
    source 7
    target 8
    bw 15
  ]
  edge [
    source 8
    target 9
    bw 50
  ]
  edge [
    source 9
    target 10
    bw 16
  ]
  edge [
    source 10
    target 11
    bw 25
  ]
  edge [
    source 11
    target 12
    bw 49
  ]
  edge [
    source 12
    target 13
    bw 13
  ]
  edge [
    source 13
    target 14
    bw 40
  ]
]
