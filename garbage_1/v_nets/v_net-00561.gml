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
  id 561
  arrival_time 10544.490177774785
  lifetime 583.4153387808408
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 1
    gpu 42
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 45
    gpu 27
    rom 47
  ]
  node [
    id 2
    label "2"
    cpu 16
    gpu 49
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 28
    gpu 41
    rom 8
  ]
  node [
    id 4
    label "4"
    cpu 45
    gpu 5
    rom 6
  ]
  node [
    id 5
    label "5"
    cpu 39
    gpu 31
    rom 44
  ]
  node [
    id 6
    label "6"
    cpu 11
    gpu 2
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 35
    gpu 21
    rom 36
  ]
  node [
    id 8
    label "8"
    cpu 3
    gpu 44
    rom 33
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 43
    rom 2
  ]
  node [
    id 10
    label "10"
    cpu 42
    gpu 1
    rom 36
  ]
  node [
    id 11
    label "11"
    cpu 18
    gpu 34
    rom 15
  ]
  node [
    id 12
    label "12"
    cpu 21
    gpu 37
    rom 24
  ]
  node [
    id 13
    label "13"
    cpu 44
    gpu 21
    rom 22
  ]
  edge [
    source 0
    target 1
    bw 17
  ]
  edge [
    source 1
    target 2
    bw 50
  ]
  edge [
    source 2
    target 3
    bw 39
  ]
  edge [
    source 3
    target 4
    bw 32
  ]
  edge [
    source 4
    target 5
    bw 19
  ]
  edge [
    source 5
    target 6
    bw 18
  ]
  edge [
    source 6
    target 7
    bw 21
  ]
  edge [
    source 7
    target 8
    bw 35
  ]
  edge [
    source 8
    target 9
    bw 32
  ]
  edge [
    source 9
    target 10
    bw 27
  ]
  edge [
    source 10
    target 11
    bw 15
  ]
  edge [
    source 11
    target 12
    bw 1
  ]
  edge [
    source 12
    target 13
    bw 25
  ]
]
